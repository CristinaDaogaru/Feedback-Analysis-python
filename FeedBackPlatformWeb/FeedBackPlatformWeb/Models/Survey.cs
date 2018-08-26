using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Web;

namespace FeedBackPlatformWeb.Models
{
    public class Survey
    {
        [Key]
        public int Id { get; set; }
        [MaxLength(20)]
        public string Name { get; set; }
        public int CategoryId { get; set; }
        public Category Category { get; set; }
        public int ClientId { get; set; }
        public ClientProfile Client { get; set; }
        //public List<Question> Questions { get; set; }
        //public List<ClientProfile> Clients { get; set; }
    }
}